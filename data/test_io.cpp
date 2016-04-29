#include "db.hpp"
#include "data.pb.h"
#include "io.hpp"
#include<string>
int main(int argc, char* argv[]) {
    // Initialize Google's logging library.
    google::InitGoogleLogging(argv[0]);

    string path="/home/yangwu/mnist/mnist_train_lmdb";

    db::DB* cifar_db = db::GetDB("lmdb");

    cifar_db->Open(path, db::READ);
    db::Cursor* cur = cifar_db->NewCursor();

    int count = 0;
    while(count < 5000) {
        Datum datum;
        datum.ParseFromString(cur->value());

        cv::Mat img;
        img = DatumToCVMat(datum);

        // Create window
        /*cv::namedWindow( "Sample Image" , cv::WINDOW_NORMAL );
        for(;;) {
            int c;
            c = cv::waitKey(10);
            if( (char)c == 27 )
            { break; }
            imshow( "Sample Image", img);
        }*/
        // imwrite(std::to_string(count) + ".jpg", img);
        count++;
        cur->Next();
    }
    cifar_db->Close();
    return 0;
}
